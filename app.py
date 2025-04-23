import os
import re
import time
import shutil
import pandas as pd
from langchain_community.llms import LlamaCpp
from datetime import date

today = date.today()
print("Today's date is:", today)


class CVProcessor:
    def __init__(self, cv_folder="cvs", archive_folder="archive", output_file="output.xlsx", interval=30):
        self.cv_folder = cv_folder
        self.archive_folder = archive_folder
        self.output_file = output_file
        self.interval = interval
        self.llm = self.load_llama_model()

        os.makedirs(self.cv_folder, exist_ok=True)
        os.makedirs(self.archive_folder, exist_ok=True)

        if not os.path.exists(self.output_file):
            self.initialize_excel()

    def load_llama_model(self):
        return LlamaCpp(
            model_path=r"C:/Users/thegh/Python Projects/Ai Models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            n_gpu_layers=10,
            n_ctx=5000,
            f16_kv=True,
            max_tokens=5000,
            temperature=0.1,
            streaming=True,
            verbose=True
        )

    def initialize_excel(self):
        columns = ["Sr No", "Candidate Name & CNIC No", "Father's Name", "DOB", "SSC Field / %age",
                   "HSSC Field / %age", "Graduation Field / CGPA / Passing Year", "Courses",
                   "Experience Detail with Dates", "Total Experience", "Contact Number",
                   "Email", "Address"]
        pd.DataFrame(columns=columns).to_excel(self.output_file, index=False)

    def extract_text_from_pdf(self, file_path):
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    def extract_info_with_llama(self, text):
        # Rough token count estimate (1 token ~ 4 characters)
        max_tokens = 4900  # keep below limit for prompt + output
        token_estimate = len(text) // 4

        if token_estimate > max_tokens:
            print("⛔ Resume too long. Truncating text to fit within context window.")
            text = text[:max_tokens * 4]  # truncate safely
        template = f"""
You are an expert resume extractor. Extract the following structured information from this resume:

1. Candidate Name & CNIC
2. Father's Name
3. Date of Birth (DOB)
4. SSC Field / %age
5. HSSC Field / %age
6. Graduation Field / CGPA / Passing Year
7. Courses
8. Experience Detail with Dates
9. Total Experience (in years and months)
10. Contact Number
11. Email
12. Address

Resume:
{text}

Required outputs (Just Mention the answer):

Candidate Name & CNIC: ... \n
Candidate Father's Name: ... \n
DOB: ... \n
SSC Field / %age: ... \n
HSSC Field / %age: ... \n
Graduation Field / CGPA / Passing Year: ... \n
Courses: ... \n
Experience Detail with Dates: ... \n
Total Experience: ... \n
Contact Number: ... \n
Email: ... \n
Address: ... \n
"""
        initial_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                            Cutting Knowledge Date: December 2023
                            Today Date: {today}

                            {template}
                            <|eot_id|><|start_header_id|>user<|end_header_id|>
                            """

        cnic_no_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Candidate Name & CNIC' From the above Resume.(Mention the Answer only)" + " Assistant:"
        cnic_no_output = self.llm.invoke(cnic_no_prompt)

        initial_general_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

                                    Cutting Knowledge Date: December 2023
                                    Today Date: {today}


                                    <|eot_id|><|start_header_id|>user<|end_header_id|>
                                    """
        father_name_prompt = initial_general_prompt + "\nUser: " + "Extract the full second name from that name : " + cnic_no_output + " (Mention the Answer only)Assistant:"
        father_name_output = self.llm.invoke(father_name_prompt)

        dob_prompt = initial_prompt + "\nUser: " + "Give me only required output 'DOB' From the above Resume.(Mention the Answer only)" + " Assistant:"
        dob_output = self.llm.invoke(dob_prompt)

        ssc_age_prompt = initial_prompt + "\nUser: " + "Give me only required output 'SSC Field / %age' From the above Resume.(Mention the Answer only)" + " Assistant:"
        ssc_age_output = self.llm.invoke(ssc_age_prompt)

        hssc_age_prompt = initial_prompt + "\nUser: " + "Give me only required output 'HSSC Field / %age' From the above Resume.(Mention the Answer only)" + " Assistant:"
        hssc_age_output = self.llm.invoke(hssc_age_prompt)

        graduation_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Graduation Field / CGPA / Passing Year' From the above Resume.(Mention the Answer only)" + " Assistant:"
        graduation_output = self.llm.invoke(graduation_prompt)

        courses_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Courses' From the above Resume.(Mention the Answer only)" + " Assistant:"
        courses_output = self.llm.invoke(courses_prompt)

        exp_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Experience Detail with Dates' From the above Resume.(Mention the Answer only)" + " Assistant:"
        exp_output = self.llm.invoke(exp_prompt)

        total_exp_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Total Experience' From the above Resume.(Mention the Answer only)" + " Assistant:"
        total_exp_output = self.llm.invoke(total_exp_prompt)

        contact_no_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Contact Number' From the above Resume.(Mention the Answer only)" + " Assistant:"
        contact_no_output = self.llm.invoke(contact_no_prompt)

        email_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Email' From the above Resume.(Mention the Answer only)" + " Assistant:"
        email_output = self.llm.invoke(email_prompt)

        address_prompt = initial_prompt + "\nUser: " + "Give me only required output 'Address' From the above Resume.(Mention the Answer only)" + " Assistant:"
        address_output = self.llm.invoke(address_prompt)

        output = {
            "Candidate Name & CNIC No": cnic_no_output,
            "Father's Name": father_name_output,
            "DOB": dob_output,
            "SSC Field / %age": ssc_age_output,
            "HSSC Field / %age": hssc_age_output,
            "Graduation Field / CGPA / Passing Year": graduation_output,
            "Courses": courses_output,
            "Experience Detail with Dates": exp_output,
            "Total Experience": total_exp_output,
            "Contact Number": contact_no_output,
            "Email": email_output,
            "Address": address_output,
        }
        # print(output)
        return output

    def append_to_excel(self, data, sr_no):
        df = pd.read_excel(self.output_file)

        # Ensure ordered keys
        ordered_keys = [
            "Candidate Name & CNIC No",
            "Father's Name",
            "DOB",
            "SSC Field / %age",
            "HSSC Field / %age",
            "Graduation Field / CGPA / Passing Year",
            "Courses",
            "Experience Detail with Dates",
            "Total Experience",
            "Contact Number",
            "Email",
            "Address"
        ]

        # Build ordered row
        row = {"Sr No": sr_no}
        for key in ordered_keys:
            row[key] = data.get(key, "")

        # Append to DataFrame
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_excel(self.output_file, index=False)
        print(f"✅ CV appended for Sr No: {sr_no}")

    # def append_to_excel(self, data, sr_no):
    #     df = pd.read_excel(self.output_file)
    #     row = {
    #         "Sr No": sr_no,
    #         **data
    #     }
    #     df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    #     df.to_excel(self.output_file, index=False)
    #     print(f"✅ CV appended for Sr No: {sr_no}")

    def process_new_cvs(self):
        files = [f for f in os.listdir(self.cv_folder) if f.lower().endswith(".pdf")]
        archived = set(os.listdir(self.archive_folder))

        if not files:
            print("📂 No CV files found.")
            return

        new_files = [f for f in files if f not in archived]
        if not new_files:
            print("🟡 No new CVs found.")
            return

        try:
            df = pd.read_excel(self.output_file)
            current_sr = len(df) + 1
        except FileNotFoundError:
            current_sr = 1

        for i, file_name in enumerate(new_files):
            file_path = os.path.join(self.cv_folder, file_name)
            print(f"🔍 Processing: {file_name}")
            text = self.extract_text_from_pdf(file_path)
            info = self.extract_info_with_llama(text)
            self.append_to_excel(info, current_sr + i)
            shutil.move(file_path, os.path.join(self.archive_folder, file_name))
            print(f"📦 Archived: {file_name}")

    def run(self):
        print("🚀 CV Processor is now running...")
        while True:
            self.process_new_cvs()
            time.sleep(self.interval)


if __name__ == "__main__":
    processor = CVProcessor()
    processor.run()