import asyncio
import sys

from llmpivot import LLMPivot, PivotConfig


async def main():
    config = PivotConfig.from_file(sys.argv[1])
    prompts = ["Explain a control group in one sentence.", "Explain random assignment in one sentence."]
    with LLMPivot(config) as llm:
        results = await asyncio.gather(*[
            llm.agenerate([{"role": "user", "content": prompt}]) for prompt in prompts
        ])
    for result in results:
        print(result["content"])


if __name__ == "__main__":
    asyncio.run(main())
