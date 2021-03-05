import pandas as pd
import matplotlib.pyplot as plt


def print_top_countries(top_length):
    df_modify = df.fillna(0).T
    df_modify['_sum'] = df_modify.sum(axis=1)
    df_modify = df_modify[df_modify['_sum'] != 0].T.drop('_sum')
    df_mean = df_modify.mean(axis=1).sort_values(ascending=False).head(top_length)
    print(df_mean)


if __name__ == "__main__":
    df = pd.read_csv('API_AG.LND.FRST.K2_DS2_en_csv_v2_2058959.csv', index_col='Country Name')
    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
    world = df.loc['World'].dropna()
    world.update(pd.Series(world.values / 1000, index=world.index))
    world.plot(style='.-', title='Мировая площадь лесов с 1990 по 2016', ylabel='Площадь, тыс. кв. км', grid=True)
    print_top_countries(45)
    top_county = ['Russian Federation', 'Brazil', 'Canada', 'United States', 'China']
    df_top = df.loc[top_county].dropna(axis=1)
    df_top.T.plot(style='.-', title='Топ 5 стран по площади лесов с 1990 по 2016', ylabel='Площадь, кв. км', grid=True)
    plt.show()
