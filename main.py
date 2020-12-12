from src.datasets_helper import DatasetsHelper

def main():
    dh = DatasetsHelper(name='Malaga')
    for i in dh.images:
        print(i)

if __name__ == '__main__':
    main()
