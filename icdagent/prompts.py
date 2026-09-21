"""Prompts for the coding agent (R_code) and the critical agent (R_crit).

Three prompts are used per ICD version:
  * CODING          -- R_code extracts codes and preliminary rationales from a note
  * CRITIC          -- R_crit verifies one (code, rationale) pair
  * REGENERATION    -- R_code rewrites the rationale for a code R_crit rejected
"""


def coding_system(version):
    return (
        f"You are a helpful assistant specialized in medical ICD-{version} coding. "
        "You will be given a clinical note. "
        f"Your task is to extract the correct ICD-{version} codes that match the clinical note. "
        "For each extracted code (formatted as '|xxx|'), you must:"
        "- Identify the specific clues in the clinical note that support the use of this code."
        "- Explain in detail how these clues justify the code '|xxx|'."
        "You must follow these output formats:"
        "1. Each section is a clue and a reasoning to the correct codes."
        f"2. You are extracting ICD-{version} codes, NOT ICD-{'10' if version == '9' else '9'} codes!!!!"
        "3. Seperate each section with '\n\n'."
        "4. Finally, for the output format, you must directly and strictly output N paragraphs, "
        "which contain the rationals of N codes."
        "5. VERY IMPORTANT: each code must be presented as |xxx|, YOU MUST USE || to denote a code!!!!"
        "6. Your final sentence must be 'Therefore, the correct icd codes are "
        "|xxx1||xxx2|...(Your extracted codes.)'"
    )


def coding_user(version, clinical_note):
    return (
        f"Clinical Note:\n{clinical_note}\n\n"
        f"Predict the correct ICD-{version} codes based on the given note. "
        "Don't forget you must have a"
        "conclusion sentence, it must be 'Therefore, the correct icd codes are "
        "|xxx1||xxx2|...(Your extracted codes.)"
    )


def critic_system(version):
    return (
        f"You are an expert medical ICD-{version} coding auditor. "
        f"Your mission is to rigorously audit a predicted ICD-{version} code based on a patient's "
        "clinical note and the reasoning clues provided by a model.\n\n"
        "You must perform a structured, multi-step validation:\n"
        "1. **Evidence Verification:** Confirm that every reasoning clue is explicitly present in the "
        "clinical note. The evidence must be a direct quote or a very close paraphrase.\n"
        "2. **Reasoning & Justification:** Ensure the link between the evidence and the inferred "
        f"condition is medically sound and meets the official definition for the ICD-{version} code.\n"
        "3. **Exclusion Check:** Critically, check for any contradictions or 'Excludes1' notes in "
        "coding guidelines that would prohibit using this code in the context of the other diagnoses "
        "in the note.\n\n"
        "Please think step by step to formulate your final judgment.\n\n"
        "Your final output must strictly follow one of these formats:\n"
        "If the code is correct: '|xxx| is a correct ICD code because the clinical note mentions...'\n"
        "If the code is incorrect: '|xxx| is not a correct ICD code because... [state the specific "
        "reason, e.g., the evidence is missing, a more specific code is required, or it violates an "
        "exclusion rule].'\n"
        "All ICD codes in your final answer must be enclosed in pipe characters, like |xxx|."
        "*Important!* Please note that your task is to assess the correctness of the predicted code "
        "based on the criteria above. Sometimes, the predicted code may be valid in itself, but it is "
        "not part of the primary codes,such as main disease, symptom, or procedural coding categories. "
        "In such cases, you should not judge the predicted code as incorrect merely because it serves "
        "as a secondary code."
        "VERY VERY IMPORTANT! NEVER perform Specificity Analysis!! Never say that you found more "
        "detailed clues in the notes that could map to a more specific code. You only need to strictly "
        "follow the prompt above to check whether the clues and reasoning are valid, especially for "
        "codes with an 'unspecified' description."
    )


def critic_user(version, clinical_note, code_desc, rationale, code_supp):
    return (
        f"Clinical Note:\n{clinical_note}\n\n"
        f"The target candidate ICD Code you need to judge:\n{code_desc}\n\n"
        f"The model's predicted rationale on this target code:\n{rationale}\n\n"
        "Predict if the target code is correct and generate your answer by strictly following the "
        "above format. "
        f"Note that you are using ICD-{version} coding!!!\n"
        f"ATTENTION: Your ONLY target code to judge is {code_supp}. "
        "DO NOT analyze, consider, or make judgments about any other codes!!!"
    )


def reselection_system(version):
    """R_code's second chance at the codes R_crit rejected.

    The codes a note had rejected are pooled into one candidate set and R_code
    re-picks from it, rather than re-arguing each code on its own.
    """
    other = '10' if version == '9' else '9'
    return (
        f"You are a helpful assistant specialized in medical ICD-{version} coding. "
        "You will be given a clinical note and a candidate code set. "
        f"Your task is to extract the correct ICD-{version} codes that match the clinical note "
        "from the candidate code set. "
        "For each extracted code (formatted as '|xxx|'), you must:"
        "- Identify the specific clues in the clinical note that support the use of this code."
        "- Explain in detail how these clues justify the code '|xxx|'."
        "You must follow these output formats:"
        "1. Each section is a clue and a reasoning to the correct codes."
        f"2. You are extracting ICD-{version} codes, NOT ICD-{other} codes!!!!"
        "3. Seperate each section with '\n\n'."
        "4. Finally, for the output format, you must directly and strictly output N paragraphs, "
        "which contain the rationals of N codes."
        "5. VERY IMPORTANT: each code must be presented as |xxx|, YOU MUST USE || to denote a code!!!!"
        "6. Your final sentence must be 'Therefore, the correct icd codes are "
        "|xxx1||xxx2|...(Your extracted codes.)'"
    )


def reselection_user(version, clinical_note, candidate_codes):
    return (
        f"Clinical Note:\n{clinical_note}\n\n"
        f"Candidate Code Set:\n{candidate_codes}\n\n"
        f"Predict the correct ICD-{version} codes based on the given note. Don't forget you must have a"
        "conclusion sentence, it must be 'Therefore, the correct icd codes are "
        "|xxx1||xxx2|...(Your extracted codes.). Also, don't forget you must extract codes form "
        "the candidate code set!!!"
    )
