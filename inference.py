from unbias_net import UnbiasNet


def main():
    net = UnbiasNet('config/config_template.yaml')
    net.inference()


if __name__ == "__main__":
    main()
