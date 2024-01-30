def get_user_input():

    list1 = list(map(int, input("Enter the elements of the first list separated by space: ").split()))
    list2 = list(map(int, input("Enter the elements of the second list separated by space: ").split()))
    return list1, list2

def find_common_elements(list1, list2):

    common_elements = set(list1) & set(list2)
    return list(common_elements)

def count_common_elements(list1, list2):

    common_elements_count = len(find_common_elements(list1, list2))
    return common_elements_count

if __name__ == "_main_":

    input_lists = get_user_input()
    list1, list2 = input_lists

    common_elements = find_common_elements(list1, list2)

    common_elements_count = count_common_elements(list1, list2)

    print(f"Common Elements: {common_elements}")
    print(f"Number of Common Elements: {common_elements_count}")