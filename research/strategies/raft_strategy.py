from strategy import Strategy


class RaftStrategy(Strategy):

    def raft(self):
        data = self.data
        buckets_in_use = 0
        entries, exits = [], []
        self.normalized('variance')
        self.normalized('volume')
        self.normalized('gap')
        self.normalized('span')
        # self.snapshot([0, 120], ['normalized_span', 'normalized_variance'])

        for index in data.index:
            row = data.iloc[index]
            print(f"{index:3d} trending {row['normalized_variance']:4d}, volume {row['normalized_volume']:3d}, tension {row['normalized_gap']:4d}, ", end="")
            print(f"span {row['normalized_span']:3d} ({row['upper_wick']*100:2.0f} {row['body_size']*100:3.0f} {row['lower_wick']*100:2.0f})", end="")

            if index > 5 and row['normalized_variance'] < -70 and (row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
                if row['lower_wick'] > 0.25:
                    print(" *****", end="")
                    entries.append(index)
            elif index > 5 and row['normalized_variance'] > 70 and (row['normalized_volume'] > 70 or row['normalized_volume'] < 30):
                if row['upper_wick'] > 0:
                    print(" ----------", end="")
                    exits.append(index)
            print()

        positions = [0] * len(data)
        for i in entries:
            positions[i] = 1
        for i in exits:
            positions[i] = -1
        data['position'] = positions

        print(entries)
        print(exits)

    def signal(self):
        self.raft()
