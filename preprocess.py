def preprocess_input(pclass, sex, age):
    sex = 0 if sex.lower() == 'male' else 1
    return [[pclass, sex, age]]
