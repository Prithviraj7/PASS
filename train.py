from unbias_net import UnbiasNet


def main():
    net = UnbiasNet('config/config_template.yaml')
    net.train()


if __name__ == "__main__":
    main()
