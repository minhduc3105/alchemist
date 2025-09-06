import warnings
warnings.filterwarnings("ignore")

import os
from config import collect_args
from executor import Executor
from labeling import Labeler
from dotenv import load_dotenv

load_dotenv()

def main():
    
    args = collect_args()
    LF_saving_parent_dir = os.path.join(args['dataset_LF_saved_path'], args['mode'])
    if not os.path.exists(LF_saving_parent_dir):
        os.mkdir(LF_saving_parent_dir)

    LF_saving_parent_dir = os.path.join(LF_saving_parent_dir, args["codellm"])
    if not os.path.exists(LF_saving_parent_dir):
        # Loại bỏ ký tự không hợp lệ trong tên thư mục (như dấu :)
        LF_saving_parent_dir = LF_saving_parent_dir.replace(":", "_")

        # Tạo toàn bộ cây thư mục nếu chưa tồn tại
        os.makedirs(LF_saving_parent_dir, exist_ok=True)

        
    py_file_count = len([f for f in os.listdir(LF_saving_parent_dir) if f.endswith('.py')])
    args["LF_saving_exact_dir"] = LF_saving_parent_dir
    args["LF_saving_parent_dir"] = LF_saving_parent_dir
    
    print(py_file_count)
    print("\n##############################")
    while py_file_count < 10:
        executor = Executor(args)
        executor.execute_mode()
        py_file_count = len([f for f in os.listdir(LF_saving_parent_dir) if f.endswith('.py')])
        print("\n##############################")
    
    labeler = Labeler(args)
    labeler.run()

if __name__ == "__main__":
    main()