# test
re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def remove_single_letter_words(text):
    words = text.split()  # Splitting the text into words
    filtered_words = [word for word in words if len(word) > 1]  # Filtering out words with length 1
    filtered_text = ' '.join(filtered_words)  # Joining the remaining words back into a string
    return filtered_text
    
    [^a-zA-Z\s]
