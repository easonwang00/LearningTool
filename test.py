from app import Generator
def demo_anthropic_query():
    agent = Generator()
    selected_text = "Analysts suggest that Tesla's direct sales model is costly and limits expansion into lower-tier cities, where competitors with dealerships like BYD have an advantage."
    language = "english"
    question = "what is tesla"
    res = agent.run_chain(language, selected_text, question)
    print(res)
    return res

def main():
    demo_anthropic_query()

if __name__ == "__main__":
    main()