import itertools
import random

insert_fingers = [
    "Insert the fingers into the drawer handle",
    "Put the fingers inside the drawer handle",
    "Place the fingers within the handle",
    "Position the fingers in the drawer handle",
    "Position the fingers to open the drawer",
    "Align the fingers with the drawer handle",
    "Slide the fingers into the handle",
    "Insert fingers into the handle of the drawer",
    "Prepare to open the drawer by inserting fingers",
    "Prepare to open the drawer inserting the fingers within the handle"
]
open_drawer = [
    "Open the drawer",
    "Pull the drawer open",
    "Slide the drawer out",
    "Open the cabinet drawer",
    "Pull the drawer towards you",
    "Open the drawer completely",
    "Can you open the drawer?",
    "Open the drawer by pulling it",
    "Can you please open the drawer?",
    "Can you slide the drawer out?",
    "Please, open the drawer completely",
    "Fully open the drawer",
    "Plese, fully open the drawer"
]

close_drawer = [
    "Close the drawer",
    "Push the drawer closed",
    "Shut the drawer",
    "Slide the drawer in",
    "Close the cabinet drawer",
    "Fully close the drawer",
    "Push the drawer back in",
    "Gently close the drawer",
    "Make sure the drawer is closed",
    "Slide the drawer shut",
    "Can you close the drawer?",
    "Please close the drawer",
    "Push the drawer all the way in",
    "Slide the cabinet drawer closed",
    "Shut the drawer completely",
    "Push the drawer back into the cabinet",
    "Bring the drawer to a closed position",
    "Close the drawer carefully",
    "Slide the drawer fully in",
    "Ensure the drawer is pushed in",
    "Return the drawer to the closed position",
    "Slide the drawer gently into place",
    "Close the drawer so it’s flush with the cabinet",
]

pick_object = [
    "Pick up the red cube",
    "Pick up the blue cube",
    "Pick up the green cube",
    "Pick up the yellow cube",
    "Pick up the black cube",
    "Pick up the purple cube",
    "Pick up the grey cube",
    "Grasp the cube",
    "Grab the object",
    "Lift the cube from the table",
    "Take the cube",
    "Pick the object up",
    "Can you grasp the cube?",
]
place_object = [
    # ---- Place on table ----
    "Place the cube on the table",
    "Put the object on the table",
    "Set the cube down on the table",
    "Drop the cube onto the table",
    "Place the block on the surface",
    "Put the cube on the tabletop",
    "Set the object on the table",
    "Carefully place the cube on the table",
    "Put the block down on the table",
    "Place the object on the flat surface",
    "Set the cube onto the table",
    "Gently put the cube on the tabletop",
    "Place the cube at the center of the table",
    "Put the cube on the left side of the table",
    "Place the cube on the right side of the table",

    # ---- Place inside drawer ----
    "Place the cube inside the drawer",
    "Put the object in the drawer",
    "Insert the cube into the drawer",
    "Place the block in the drawer",
    "Put the cube inside the cabinet",
    "Slide the cube into the drawer",
    "Carefully place the object in the drawer",
    "Put the block gently inside the drawer",
    "Insert the object fully into the drawer",
    "Place the cube completely inside the drawer",
    "Put the object into the drawer without spilling",
    "Slide the cube carefully into the cabinet",

    # ---- Stack cubes ----
    "Stack the red cube on top of the yellow cube",
    "Place the red cube on the yellow one",
    "Put one cube on top of another cube",
    "Stack one cube on another",
    "Place the cube on top of the other cube",
    "Stack the blue cube on the green cube",
    "Put the green cube on the purple cube",
    "Stack the yellow cube on the red cube",
    "Place the top cube carefully on the bottom cube",
    "Put the block on top of another block",
    "Stack the cube gently on top of another cube",
    "Place the cube precisely on the other cube",
    "Stack two cubes on each other",
    "Put the upper cube on the lower cube",
    "Carefully stack the red block on the blue block",
    "Place one cube onto another with precision",

    # ---- Mixed phrasing / generic ----
    "Place the object on the table or surface",
    "Put the cube somewhere on the tabletop",
    "Insert the cube into a drawer",
    "Stack one block on another",
    "Place the block either on top of another block or in the drawer",
    "Carefully put the object where it belongs",
    "Place the cube at the correct position",
    "Stack the cubes according to the instructions",
    "Put the cube down safely on the table",
    "Place the block inside the storage area",
    "Put the object neatly on the table",
]


def make_pairs(prompts, max_pairs=None):
    pairs = list(itertools.combinations(prompts, 2))
    random.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


def generate_training_pairs():
    train_pairs = []

    train_pairs += make_pairs(insert_fingers, max_pairs=50)
    train_pairs += make_pairs(open_drawer, max_pairs=50)
    train_pairs += make_pairs(close_drawer, max_pairs=50)
    train_pairs += make_pairs(pick_object, max_pairs=50)
    train_pairs += make_pairs(place_object, max_pairs=50)

    print(f"Total training pairs: {len(train_pairs)}")

    return train_pairs
