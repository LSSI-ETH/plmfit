import random
import tqdm


def get_payout(spin):
    """
    Returns the payout for a given spin (3 symbols).
    Modify this function to define your own custom payout rules.
    """
    s1, s2, s3 = spin

    # Example payout rules (customize to your liking):
    # ------------------------------------------------
    #  3 of a kind:
    #   - 7,7,7 => 2
    #   - cherries,cherries,cherries => 10
    #   - grapes,grapes,grapes => 40
    #   - bell,bell,bell => 400
    #   - BAR,BAR,BAR => 500
    # ------------------------------------------------

    if s1 == s2 == s3:
        if s1 == "7":
            return 1.5
        elif s1 == "cherries":
             return 10
        elif s1 == "grapes":
             return 30
        elif s1 == "bell":
             return 100
        elif s1 == "BAR":
             return 1000
    elif s1 == s2:
        if s1 == "7":
            return 0.25
        elif s1 == "cherries":
             return 2
        elif s1 == "grapes":
             return 4
        elif s1 == "bell":
             return 10
        elif s1 == "BAR":
             return 0
    return 0



def simulate_slot_machine(bankroll, bet_size=1):
    """
    Simulate the slot machine until we spend 'bankroll' dollars.
    Each spin costs 'bet_size' dollars. Return the total amount won and the RTP.
    """

    # Define each reel's distribution of symbols (3 reels, same distribution)
    symbols = ["7"] * 8 + ["cherries"] * 4 + ["grapes"] * 2 + ["bell"] * 1 + ["BAR"] * 1

    # We’ll calculate how many spins we can afford:
    num_spins = bankroll // bet_size

    total_spent = num_spins * bet_size
    total_won = 0

    for _ in tqdm.tqdm(range(num_spins)):
        # Randomly pick 1 symbol from the distribution for each of the 3 reels
        spin_result = (
            random.choice(symbols),
            random.choice(symbols),
            random.choice(symbols),
        )

        # Calculate payout for this spin
        payout = get_payout(spin_result)
        total_won += payout

    rtp = total_won / total_spent if total_spent > 0 else 0
    return total_spent, total_won, rtp


def main():
    # Example: Simulate spending 1,000,000,000 dollars at $1 per spin
    # WARNING: This may be very slow in plain Python!
    bankroll = 10000000  # 10 million
    bet_size = 1

    print("Starting simulation...")
    spent, won, rtp = simulate_slot_machine(bankroll, bet_size)

    print(f"Total spent: ${spent}")
    print(f"Total won:  ${won}")
    print(f"RTP:        {rtp * 100:.2f}%")


if __name__ == "__main__":
    main()
