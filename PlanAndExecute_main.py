from PlanAndExecute import CommandAgent


def main():
    # Load OpenAI API key
    with open('apikey.txt', 'r') as file:
        openai_key = file.read().replace('\n', '')
    
    

    agent = CommandAgent(openai_key)
    while True:
        agent.run()
        

if __name__ == "__main__":
    main()