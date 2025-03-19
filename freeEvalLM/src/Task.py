import glob, os
current_path = os.getcwd()

from freeEvalLM._lib._df import read_file


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
            self.data_path = "datasets/mmlu_pro_0shots"
        elif self.task == "ifeval":
            self.data_path = os.path.join("datasets/ifeval")
        elif self.task == "livebench":
            self.data_path = os.path.join("datasets/livebench")
        elif self.task == "strong_reject":
            self.data_path = os.path.join("datasets/strong_reject")
        elif self.task == "wild_jailbreak":
            self.data_path = os.path.join("datasets/wild_jailbreak")
        elif self.task == "XSTest_S":
            self.data_path = os.path.join("datasets/XSTest_S")
        else:
            print("self.task is", self.task)


         
    def load_dataset(self):
        print("loading dataset")
        print(self.data_path,)
        subtasks_data_path = glob.glob(os.path.join(self.data_path, "*.json")) + glob.glob(os.path.join(self.data_path, "*.csv"))
        all_dfs = []
        print(subtasks_data_path)

        for data_path in subtasks_data_path:
            print("loading subtask: ", os.path.basename(data_path).split(".")[0])
            subtask_name = os.path.basename(data_path).split(".")[0]
            subtask_data = read_file(data_path)

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
            from freeEvalLM.tasks.mmlu_pro.mmlu_pro import mmlu_pro as Evaluator
        elif self.task == "ifeval":
            from freeEvalLM.tasks.ifeval.ifeval import ifeval as Evaluator
        elif self.task == "livebench":
            from freeEvalLM.tasks.livebench.livebench import livebench as Evaluator
        elif self.task == "wild_jailbreak":
            from freeEvalLM.tasks.asr.wild_jailbreak import asr as Evaluator
        elif self.task == "XSTest_S":
            from freeEvalLM.tasks.asr.XSTest import asr as Evaluator
        elif self.task == "strong_reject":
            from freeEvalLM.tasks.strong_reject.strong_reject import strong_reject as Evaluator
        else:
            from freeEvalLM.src.Evaluator import Evaluator as Evaluator

        evaluator = Evaluator(self.task, self.save_path)
        return evaluator


if __name__ == "__main__":
    a = TaskLoader("mmlu_pro")
    a.load()






