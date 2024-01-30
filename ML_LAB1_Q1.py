no_of_vowels = 0
no_of_consonants = 0

def findVowels(input):
    vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
    
    for x in input:
        if x in vowels:
            global no_of_vowels
            no_of_vowels += 1
        else:
            global no_of_consonants
            no_of_consonants += 1
    return no_of_vowels, no_of_consonants

def main():
    input_str = input("Enter a Word or a Sentence: ")
    no_of_vowels, no_of_consonants = findVowels(input_str)
    print("No of vowels in given input: ", no_of_vowels)
    print("No of consonants in given input: ", no_of_consonants)

if __name__ == "__main__":
    main()
