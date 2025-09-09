import os
import torch
from openai import OpenAI
from anthropic import Anthropic
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import custom_input
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, QWEN_API_KEY
from modes.base_mode import BaseMode  


class ScriptoriumWSMode(BaseMode):
    
    def __init__(self, args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature):
        super().__init__(args, codellm_system_prompt, mission_statement, labeling_instruction, function_signature)
        self.codellm_choice = self.args["codellm"]
        self.final_prompt = None
        self.synthesized_labeling_function = None
        self.estimated_cost = 0

        # ✅ Nếu là qwen_local thì load model HuggingFace
        if self.codellm_choice.startswith("qwen-local"):
            model_id = "Qwen/Qwen2-VL-7B-Instruct"   # ✅ chọn bản nhẹ
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                device_map="auto"
            )

    
    def _load_qwen_local(self, model_name: str):
        print(f"Loading Qwen model from Hugging Face: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def gpt_inference_call(self):
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=self.codellm_choice,
            temperature=1.0,
            max_tokens=20000,
            messages=[
                {"role": "system", "content": self.codellm_system_prompt},
                {"role": "user", "content": self.final_prompt}
            ]
        )
        return completion

    def claude_inference_call(self):
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        completion = client.messages.create(
            model=self.codellm_choice,
            temperature=0.7,
            max_tokens=1000,
            system=self.codellm_system_prompt,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": self.final_prompt}]}
            ]
        )
        return completion


    def qwen_local_inference_call(self):
        if not self.final_prompt:
            raise ValueError("⚠️ final_prompt is empty!")

        # nối system + user prompt
        prompt = (self.codellm_system_prompt or "") + "\n" + self.final_prompt

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



    def run(self):
        self.final_prompt = (
            self.mission_statement
            + "\n" + self.labeling_instruction
            + "\n" + self.function_signature
        )
        
        print(f"\nFinal Prompt for CodeLLM ({self.codellm_choice}): \n'''{self.final_prompt}'''\n")
        print("##############################\n")

        if self.codellm_choice in ["gpt-3.5-turbo", "gpt-4.1"]:
            completion = self.gpt_inference_call()
            self.calculate_cost(
                input_tok=completion.usage.prompt_tokens,
                output_tok=completion.usage.completion_tokens,
                model=self.codellm_choice
            )
            response = completion.choices[0].message.content

        elif self.codellm_choice in ["claude-2.1", "claude-3-sonnet-20240229"]:
            completion = self.claude_inference_call()
            self.calculate_cost(
                input_tok=completion.usage.input_tokens,
                output_tok=completion.usage.output_tokens,
                model=self.codellm_choice
            )
            response = completion.content[0].text

        elif self.codellm_choice.startswith("qwen-local"):
            response = self.qwen_local_inference_call()
        
        else:
            raise NotImplementedError(f"Model {self.codellm_choice} chưa hỗ trợ.")

        self.synthesized_labeling_function = self.extract_LF(response)

        if self.synthesized_labeling_function is None:
            return None

        print(f"\nYour synthesized labeling function is:\n{self.synthesized_labeling_function}\n")
        print("##############################\n")

        if self.is_runnable_lf(self.synthesized_labeling_function):
            self.save()

        return None

    @staticmethod
    def is_runnable_lf(code_str: str, test_input=None) -> bool:
        try:
            compile(code_str, '<string>', 'exec')
        except Exception as e:
            print(f"LF Syntax Error: {e}")
            return False

        try:
            exec_env = {}
            exec(code_str, exec_env, exec_env)
            lf_func = exec_env.get("label_function")
            if not callable(lf_func):
                print("No callable function named 'label_function' found.")
                return False
            if test_input is not None:
                lf_func(test_input)

        except Exception as e:
            print(f"LF Runtime Error: {e}")
            return False

        return True
