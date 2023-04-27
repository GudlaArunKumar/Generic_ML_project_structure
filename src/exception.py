import sys 
import logging 
import logger

"""
Module to return a error message based on defined custom exception class
"""

# sys contains details of the exception and its traceback

def error_message_details(error, error_detail:sys):

    _, _, exec_tb = error_detail.exc_info() # we will take traceback of error
    file_name = exec_tb.tb_frame.f_code.co_filename 
    error_message = "Error occured in python script name [{0}] on line number [{1}] and error message [{2}]".format(file_name, 
    exec_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    # custom exception will be called in all defined modules in the package

    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        """
        Returns error message when custom exception is called
        """
        return self.error_message
    


