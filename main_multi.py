import warnings
warnings.filterwarnings("ignore")

import os
from config import collect_args
from executor import Executor
from labeling import Labeler
from dotenv import load_dotenv

load_dotenv()

def main():
    label_folders = []
    path = "data_home/pubmed"
    
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            label_folders.append(folder)
    
    for label_num in range(len(label_folders)):
        args = collect_args()
        LF_saving_parent_dir = os.path.join(args['dataset_LF_saved_path'], str(label_num), args['mode'])
        
        args['exp_result_saved_path'] = os.path.join(args['exp_result_saved_path'], str(label_num))
            
        args["prompt_template"] = args["prompt_template"][label_num]
        
        if not os.path.exists(LF_saving_parent_dir):
            os.mkdir(LF_saving_parent_dir)

        LF_saving_parent_dir = os.path.join(LF_saving_parent_dir, args["codellm"])
        if not os.path.exists(LF_saving_parent_dir):
            os.mkdir(LF_saving_parent_dir)
            
        if not os.path.exists(args['exp_result_saved_path']):
            os.mkdir(args['exp_result_saved_path'])
            
        args['LF_saving_parent_dir'] = LF_saving_parent_dir
        args['LF_saving_exact_dir'] = LF_saving_parent_dir
            
        py_file_count = len([f for f in os.listdir(LF_saving_parent_dir) if f.endswith('.py')])
        
        print("\n##############################")
        while py_file_count < 15:
            executor = Executor(args)
            executor.execute_mode()
            py_file_count = len([f for f in os.listdir(LF_saving_parent_dir) if f.endswith('.py')])
            print("\n##############################")
        
        labeler = Labeler(args)
        labeler.run(label_num=label_num)

if __name__ == "__main__":
    main()