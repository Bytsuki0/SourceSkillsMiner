import csv
import random
import json

# Define typical language sets per programmer type
programmer_profiles = {
    "Frontend": ["HTML", "CSS", "JavaScript", "TypeScript", "React", "Vue", "Sass"],
    "Backend": ["Python", "Java", "Go", "PHP", "C#", "Ruby", "SQL", "Node.js"],
    "DevOps": ["Go", "Shell", "Dockerfile", "HCL", "Python", "YAML", "Groovy"],
    "DataScientist": ["Python", "R", "SQL", "Julia", "Scala", "MATLAB"],
    "FullStack": ["HTML", "CSS", "JavaScript", "Python", "Go", "PHP", "SQL"],
    "Mobile": ["Kotlin", "Swift", "Java", "Dart", "JavaScript"],
    "GameDev": ["C++", "C#", "Python", "Lua", "Rust"],
    "Automation": ["Python", "Shell", "PowerShell", "Go", "Ruby", "Batch"],
    "DataEngineer": ["Python", "SQL", "Scala", "Go", "Bash", "Java"],
    "CyberSecurity": ["Python", "C", "C++", "Go", "Shell", "PHP", "JavaScript"]
}

def generate_language_profile(langs, base_scale=1.0):
    """Generate a fake language usage dictionary"""
    n_langs = random.randint(4, min(8, len(langs)))
    selected = random.sample(langs, n_langs)
    profile = {}
    for lang in selected:
        lines = int(random.randint(500, 500_000) * base_scale * random.uniform(0.5, 2.5))
        repos = random.randint(1, 300)
        profile[lang] = [lines, repos]
    return profile

# Create CSV
with open("programmer_profiles.json.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["profile", "programmer_type"])

    for _ in range(10_000):
        dev_type = random.choice(list(programmer_profiles.keys()))
        langs = programmer_profiles[dev_type]
        scale = random.uniform(0.8, 1.5)
        profile = generate_language_profile(langs, scale)
        profile_str = json.dumps(profile)
        writer.writerow([profile_str, dev_type])

print("✅ programmer_profiles.json.csv successfully created!")
