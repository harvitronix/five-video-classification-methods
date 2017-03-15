"""
Try to "classify" samples based on random chance and always guessing
the most popular category.
"""
import random
from data import DataSet

most_pop = 'TennisSwing'

data = DataSet()
nb_classes = len(data.classes)

# Try a random guess.
nb_random_matched = 0
nb_mode_matched = 0
for item in data.data:
    choice = random.choice(data.classes)
    actual = item[1]

    if choice == actual:
        nb_random_matched += 1

    if actual == most_pop:
        nb_mode_matched += 1

random_accuracy = nb_random_matched / len(data.data)
mode_accuracy = nb_mode_matched / len(data.data)
print("Randomly matched %.2f%%" % (random_accuracy * 100))
print("Mode matched %.2f%%" % (mode_accuracy * 100))
