"""
@Authors  : Claudio Jardim (CJ)
@Contact  : claudiomj8@gmail.com
@License  :
@Date     : 6 April 2022
@Version  : 0.1
@Desc     : General useful functions
"""

def read_list(file_name: str):
    with open(file_name, "r") as file:
        lines = file.read().split("\n")
        return lines

def save_list(file_name: str, list: list):
    with open(file_name, "w") as file:
        file.write("\n".join(list))