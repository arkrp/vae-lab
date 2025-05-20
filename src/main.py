import logging
def main():
    logging.basicConfig(filename=".log", filemode='w', level=logging.INFO)
    import Model.NonUniformEncoder
    import Model.Decoder
if __name__=="__main__":
    main()
