# Leetcode

# 10/28/2019
# 208. Implement Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False

class Trie:
    """
    key is to when moving on to next letter, make sure to move node pointer as well
    """
    def __init__(self):
        """
        Initialize your data structure here.
        """
        # get dictionary so platform is ready
        self.root = TrieNode()


    def insert(self, word):
        """
        Inserts a word into the trie.
        algorithm
        1, get root first and don't move its pointer
        2, create empty dictionary -> check if current letter is already in dictionary
            if not, put current letter in dictionary, and move node to next dictionary
            just like tree
        3, dictionary gets another dictionary
        """
        # Time complexity: O(m), where m is the key length.
        # get the root like LinkedList
        node = self.root
        for letter in word:
            # if letter is not in current node's dictionary, current letter get new dictionary
            if letter not in node.children:# TypeError: argument of type 'TrieNode' is not iterable
                node.children[letter] = TrieNode()

            # node pointer moves to empty dictionary so next word can go into dictionary
            node = node.children[letter]

        # empty dictionary at end will get its flag true
        node.word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        leaf holds true so if word can reach leaf, that word exists in tree
        """
        # Time complexity: O(m) In each step of the algorithm we search for the next key character.
        # start from the top
        node = self.root
        # check each letter
        for letter in word:
            if letter not in node.children:
                return False

            node = node.children[letter]
        return node.word

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        # Time complexity : O(m)
        node = self.root
        for letter in prefix:
            # if letter does not exist in trie, false
            if letter not in node.children:
                return False

            node = node.children[letter]

        # all letter in prefix exist, return True
        return True


# Your Trie object will be instantiated and called as such:
trie = Trie();
trie.insert("apple");
print(trie.search("apple"));  # // returns true
print(trie.search("app"));  # // returns false
print(trie.startsWith("app"));  # // returns true


