from fastapi import FastAPI

def generate_api() -> FastAPI:
    print("generate_api() is running!")
    return FastAPI()
