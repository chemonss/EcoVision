""" 
Class mapping utilities for EcoVision. 

This module defines the mapping between original TACO dataset categories 
and the coarse waste classes used in the project: 
plastic, paper/cardboard, metal, glass, organic, and other. 
"""

from __future__ import annotations

COARSE_CLASSES = [
    "rigid_plastic",
    "soft_plastic",
    "paper/cardboard",
    "metal",
    "glass",
    "other",
]

COARSE_CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(COARSE_CLASSES)
}

ID_TO_COARSE_CLASS = {
    class_id: class_name
    for class_name, class_id in COARSE_CLASS_TO_ID.items()
}


def map_taco_category_to_coarse(name: str, supercategory: str | None = None) -> str:
    """
    Map an original TACO category to one of the EcoVision coarse classes.

    Parameters
        name:
            Original TACO category name.

    Returns
        str
            One of: plastic, paper/cardboard, metal, glass, organic, other.
    """
    text = f"{name} {supercategory or ''}".lower()

    # Glass
    if any(keyword in text for keyword in ["glass", "jar"]):
        return "glass"

    # Metal
    if any(
        keyword in text
        for keyword in [
            "metal",
            "aluminium",
            "aluminum",
            "can",
            "foil",
            "aerosol",
            "pop tab",
            "scrap metal",
        ]
    ):
        return "metal"

    # Paper / cardboard
    if any(
        keyword in text
        for keyword in [
            "paper",
            "cardboard",
            "carton",
            "pizza box",
            "egg carton",
            "toilet tube",
            "magazine",
            "tissues",
            "paper bag",
            "paper straw",
        ]
    ):
        return "paper/cardboard"

    # Soft plastic: bags, wrappers, films, flexible packaging
    if any(
        keyword in text
        for keyword in [
            "wrapper",
            "film",
            "plastic bag",
            "garbage bag",
            "crisp packet",
            "six pack rings",
            "plastic bag & wrapper",
            "squeezable tube",
        ]
    ):
        return "soft_plastic"
    
    # Rigid plastic: bottles, caps, lids, containers, cups, utensils
    if any(
        keyword in text
        for keyword in [
            "plastic bottle",
            "bottle cap",
            "plastic lid",
            "lid",
            "plastic container",
            "tupperware",
            "plastic cup",
            "plastic utensils",
            "utensils",
            "plastic straw",
            "straw",
            "styrofoam",
            "foam",
            "polypropylene",
            "plastic",
        ]
    ):
        return "rigid_plastic"

    return "other"


def map_taco_category_to_coarse_id(name: str, supercategory: str | None = None) -> int:
    """
    Map an original TACO category to the numeric ID of the EcoVision class.
    """
    coarse_class = map_taco_category_to_coarse(name, supercategory)
    return COARSE_CLASS_TO_ID[coarse_class]

