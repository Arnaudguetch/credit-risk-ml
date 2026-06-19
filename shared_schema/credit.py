from pydantic import BaseModel, Field 

class CreditRequest(BaseModel): 
    age: int = Field(gt=0, lt=120)
    sex: str 
    job: int 
    housing: str 
    saving_accounts: str 
    checking_account: str 
    credit_amount: float = Field(gt=0) 
    duration: int = Field(gt=0)
    purpose: str 
    

model_config = {
    "extra": "forbid"
}
    
    
class CreditResponse(BaseModel):
    default_probability: float 
    prediction: int 
    risk_label: str 
    
