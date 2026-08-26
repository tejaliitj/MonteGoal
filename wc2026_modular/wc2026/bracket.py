"""
bracket.py -- the real fixed FIFA Round-of-32 structure and the real fixed
progression graph all the way to the final.

R32_FIXTURES defines each of the 16 R32 matches by ROLE, not by team name:
    "W:<letter>" = that group's winner
    "R:<letter>" = that group's runner-up
    "T:<letter>" = the third-place qualifier assigned to that slot letter
(the specific team names get resolved in tournament.py once group results
are known).

R16_PAIRS / QF_PAIRS / SF_PAIRS define which match's winner plays which
other match's winner in each subsequent round -- this is fixed and doesn't
depend on team identity at all, only on match position.

NOTE on third-place slot assignment: FIFA's exact rule for which specific
third-placed team fills each "T:<letter>" slot depends on a published
495-row combination table (which 8 of the 12 groups' third-place teams
qualified). Embedding that whole table verbatim was impractical, so
assign_third_place_slots() below uses a documented simplified rule instead.
Everything else in this module (who plays whom by role, and the match graph
through R16/QF/SF/Final) is the real FIFA structure.
"""

R32_FIXTURES = [
    ("R:A", "R:B"),   # Match 73
    ("W:E", "T:E"),   # Match 74
    ("W:F", "R:C"),   # Match 75
    ("W:C", "R:F"),   # Match 76
    ("W:I", "T:I"),   # Match 77
    ("R:E", "R:I"),   # Match 78
    ("W:A", "T:A"),   # Match 79
    ("W:L", "T:L"),   # Match 80
    ("W:D", "T:D"),   # Match 81
    ("W:G", "T:G"),   # Match 82
    ("R:K", "R:L"),   # Match 83
    ("W:H", "R:J"),   # Match 84
    ("W:B", "T:B"),   # Match 85
    ("W:J", "R:H"),   # Match 86
    ("W:K", "T:K"),   # Match 87
    ("R:D", "R:G"),   # Match 88
]

# R16 pairs index into the 16 R32 winners (0=M73 .. 15=M88).
R16_PAIRS = [
    (1, 4),    # M74 vs M77  -> "89"
    (0, 2),    # M73 vs M75  -> "90"
    (3, 5),    # M76 vs M78  -> "91"
    (6, 7),    # M79 vs M80  -> "92"
    (10, 11),  # M83 vs M84  -> "93"
    (8, 9),    # M81 vs M82  -> "94"
    (13, 15),  # M86 vs M88  -> "95"
    (12, 14),  # M85 vs M87  -> "96"
]

# QF pairs index into the 8 R16 winners (0="89"..7="96").
QF_PAIRS = [
    (0, 1),  # "89" vs "90" -> "97"
    (4, 5),  # "93" vs "94" -> "98"
    (2, 3),  # "91" vs "92" -> "99"
    (6, 7),  # "95" vs "96" -> "100"
]

# SF pairs index into the 4 QF winners (0="97".."3"="100").
SF_PAIRS = [
    (0, 1),  # "97" vs "98" -> "101"
    (2, 3),  # "99" vs "100" -> "102"
]


def assign_third_place_slots(qualifying_groups):
    """Simplified substitute for FIFA's full 495-combination table (see
    module docstring). Sorts qualifying third-place groups alphabetically
    and assigns them in order to the eight receiving slots, swapping to
    avoid a group's own third-place team facing its own group's winner
    where possible. Returns dict: slot_letter -> group_letter."""
    sorted3 = sorted(qualifying_groups)
    slot_order = ["E", "I", "A", "L", "D", "G", "B", "K"]

    slot_to_third_group = {}
    for i, slot in enumerate(slot_order):
        if i < len(sorted3):
            slot_to_third_group[slot] = sorted3[i]

    for i, slot in enumerate(slot_order):
        if slot_to_third_group.get(slot) == slot:
            for j, other_slot in enumerate(slot_order):
                if j == i:
                    continue
                if (slot_to_third_group.get(other_slot) != slot and
                        slot_to_third_group.get(other_slot) != other_slot):
                    slot_to_third_group[slot], slot_to_third_group[other_slot] = \
                        slot_to_third_group[other_slot], slot_to_third_group[slot]
                    break
    return slot_to_third_group


def resolve_role(role, group_winner, group_runner_up, third_team_by_group, slot_to_third_group):
    """Turns a role string like 'W:A' / 'R:B' / 'T:E' into an actual team name."""
    kind, letter = role[0], role[2:]
    if kind == "W":
        return group_winner[letter]
    elif kind == "R":
        return group_runner_up[letter]
    else:  # 'T'
        return third_team_by_group[slot_to_third_group[letter]]
