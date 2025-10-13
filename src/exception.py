import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line = exc_tb.tb_lineno
    error_message = f"Error in [{file_name}] at line [{line}]: {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self, message, error_detail: sys = None):
        super().__init__(message)
        self.error_detail = error_detail

    def __str__(self):
        return f"CustomException: {self.args[0]}"