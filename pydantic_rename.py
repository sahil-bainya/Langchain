from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10,default=5,description="A decimal field that represents the cgpa of the student")

# newStudent = {'name':45} --> error - Input should be a valid string
# newStudent = {'name':'sahil','age':'32'}  --> pydantic will convert '32' to 32 , str to int if possible
# newStudent = {"name": "abc", "email": "abc"} --> error -value is not a valid email address
# newStudent = {"name": "abc", "email": "abc@yahoo.com", "cgpa": 12} --> error - Input should be less than 10
newStudent = {"name": "abc", "email": "abc@yahoo.com", "cgpa": 9.5} # if 9 then it will convert it into 9.0

student = Student(**newStudent)

print(student)

#pydantic object can be converted into json or dict
student_dict = dict(student) 
student_json = student.model_dump_json() 