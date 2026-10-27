
from research_agent.agent import run_research


def main():
    print("=== Industry Intelligence Agent ===")

    query = input(
        "\nEnter the job role and market to research: "
    ).strip()

    if not query:
        print("Error: Research query cannot be empty.")
        return

    try:
        report = run_research(query)

        print("\n" + "=" * 60)
        print("INDUSTRY INTELLIGENCE REPORT")
        print("=" * 60)
        print(report)

    except Exception as exc:
        print(f"\nResearch failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
