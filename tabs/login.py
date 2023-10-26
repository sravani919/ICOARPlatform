def verify_email(email):
    import os

    path = "./data/"
    print("in verify email")
    if os.path.exists(os.path.join(path, email)):
        print("User exists")
    else:
        os.mkdir(path + email)
        print("dir created", path + email)
