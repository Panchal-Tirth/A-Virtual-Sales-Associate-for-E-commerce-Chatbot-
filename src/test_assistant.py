from assistant import Assistant

def run(title, turns):
    print(f"\n{'=' * 70}\n{title}")
    a = Assistant()
    for t in turns:
        out = a.reply(t)
        print(f"\nUSER: {t}\nBOT [{out['kind']}]: {out['text']}")
        for i, m in enumerate(out["items"], 1):
            print(f"   [{i}] {m['title'][:50]} | {m['brand']} | ₹{m['price']} | ★{m['rating']} | {m['group']}")
        if out["table"]: print("   TABLE:", out["table"][0])

run("Goa shirts + refinement",   ["I am going to Goa next month. Suggest some black shirts.", "show me cheaper ones", "compare 1 and 2"])
run("Vague request + follow-up", ["I need something to wear", "something for a college fest"])
run("Gym bundle",                ["I am joining a gym next week. What shoes and clothes should I buy? Budget around 3000"])
run("Budget tiers",              ["Show me jeans in different budgets, around 1500"])
run("Out of scope",              ["Suggest the best camera phone under 40000", "dresses for my wife"])