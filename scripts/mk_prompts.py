import json, random, sys
out = sys.argv[1]
random.seed(7)
names = ["Ayse","Mehmet","Zeynep","Ali","Fatma","Can","Elif","Burak","Deniz","Selin","Emre","Gul"]
cities = ["Ankara","Izmir","Bursa","Antalya","Konya","Adana","Trabzon","Samsun"]
recs = []
for i in range(1, 19):
    recs.append({"id": 1000+i, "name": random.choice(names), "city": random.choice(cities),
                 "status": random.choice(["pending","pending","active"]),
                 "tags": random.sample(["vip","new","late","remote","trial","paid"], 2),
                 "balance": round(random.uniform(10, 5000), 2)})
doc = json.dumps({"customers": recs}, indent=2)
sysmsg = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
copy = (sysmsg + "<|im_start|>user\nHere is a JSON document:\n```json\n" + doc + "\n```\n"
        "Rewrite the document exactly as it is, but change every \"status\": \"pending\" to \"status\": \"done\". "
        "Output only the JSON.<|im_end|>\n<|im_start|>assistant\n")
essay = (sysmsg + "<|im_start|>user\nWrite a detailed, factual essay about the history of computing, "
         "from mechanical calculators to modern GPUs.<|im_end|>\n<|im_start|>assistant\n")
open(out + "/p_copy.txt", "w").write(copy)
open(out + "/p_essay.txt", "w").write(essay)
print(len(copy), len(essay))
