import glob, os
current_path = os.getcwd()

from _lib._df import read_file


class TaskLoader:
    def __init__(self, 
        task, 
        save_path,
        sample: int = -1
        ):
         self.task = task
         self.save_path = save_path
         self.sample = sample
         self.doc()
         
    def doc(self):
        # print(self.task)
        if self.task == "mmlu_pro":
            self.data_path = "/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/datasets/mmlu_pro_0shots"
        elif self.task == "ifeval":
            self.data_path = os.path.join("/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/datasets/ifeval")
        elif self.task == "livebench":
            self.data_path = os.path.join("/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/datasets/livebench")
        else:
            print("self.task is", self.task)

            
        # print(self.data_path)
         
    def load_dataset(self):
        print("loading dataset")
        print(self.data_path,)
        subtasks_data_path = glob.glob(os.path.join(self.data_path, "*.json"))
        all_dfs = []
        print(subtasks_data_path)
        # subtasks_data_path = ["/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/datasets/mmlu_pro_0shots/engineering.json"]
        for data_path in subtasks_data_path:
            print("loading subtask: ", os.path.basename(data_path).split(".")[0])
            subtask_name = os.path.basename(data_path).split(".")[0]
            subtask_data = read_file(data_path)

            # if self.sample > 0:
            #     subtask_data = subtask_data.sample(n=self.sample, random_state=1234).reset_index(drop=True)
            if self.sample > 0:
                subtask_data = subtask_data.tail(self.sample).reset_index(drop=True)

            _dict = {
                "subtask_name": subtask_name,
                "subtask_data": subtask_data
            }

            all_dfs.append(_dict)
        
        return all_dfs
            
    def load_evaluator(self):
        if self.task == "mmlu_pro":
            from tasks.mmlu_pro.mmlu_pro import mmlu_pro as Evaluator
        if self.task == "ifeval":
            from tasks.ifeval.ifeval import ifeval as Evaluator
        if self.task == "livebench":
            from tasks.livebench.livebench import livebench as Evaluator

        evaluator = Evaluator(self.task, self.save_path)
        return evaluator


if __name__ == "__main__":
    a = TaskLoader("mmlu_pro")
    a.load()






