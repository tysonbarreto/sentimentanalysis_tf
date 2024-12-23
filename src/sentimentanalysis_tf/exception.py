import sys

class ml_exception(Exception):


    def __init__(self, error_message, error_detail:sys):
        self.error_message = error_message
        _,_,exec_tb=error_detail.exc_info()

        self.lineno = exec_tb.tb_lineno
        self.filename = exec_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in python script name {self.file_name} line number {self.lineno} error message {str(self.error_message)}"
    
if __name__=="__main__":
    __all__=["ml_exception"]