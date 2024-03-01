import json

def get_template_db():
	file = open('tools/ai_template.json','rb')
	db = json.load(file)
	return db
