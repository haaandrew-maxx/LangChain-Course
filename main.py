from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

if __name__ == "__main__":
    print("This is the main module.")
    print(app.invoke(input={"question" : "what is a pizza"}))
