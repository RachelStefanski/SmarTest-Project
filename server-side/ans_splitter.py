from wtpsplit import SaT
sat = SaT("sat-12l-sm")

def split_text(text_to_split):
    semantic_units = sat.split(text_to_split)
    return semantic_units

if (__name__ == "__main__"):
    # Example usage
    text = "The Blitz was a sustained bombing campaign by Nazi Germany against Britain, especially London, from 1940 to 1941. Its effect on civilians included thousands of deaths, destruction of homes, fear, and the evacuation of children to the countryside.	"



    split_units = split_text(text)
    print(split_units)  # Output: ['This is a test.', 'This is another test.']