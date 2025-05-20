import logging
def main():
    logging.basicConfig(filename="log.log", filemode='w', level=logging.INFO)
    import Model.Encoder
    import Model.NonUniformEncoder
    import Model.Decoder
if __name__=="__main__":
    main()
