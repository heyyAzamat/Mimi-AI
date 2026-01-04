import google.generativeai as genai

genai.configure(api_key="AIzaSyBgBwnQGga_HqsdW9BJWOQQN-MUyuEtaW4")
model = genai.GenerativeModel("gemini-pro")

response = model.generate_content("Say hello")
print(response.text)
