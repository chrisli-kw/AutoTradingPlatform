from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

# ssh-keygen -P "" -t rsa -b 4096 -m pem -f sinopac
# ssh-keygen - f sinopac.pub -e -m pem > sinopac_public_key.pem


class CipherTool:
    def __init__(self, decrypt: bool = True, encrypt: bool = True, **kwargs):
        self.DECRYPT = decrypt
        self.ENCRYPT = encrypt
        self.__load_keys(**kwargs)

    def __load_keys(self, **kwargs):
        key_path = kwargs['key_path'] if 'key_path' in kwargs else "./lib/ckey"
        if self.ENCRYPT:
            try:
                with open(f"{key_path}/sinopac_public.pem", "rb") as f:
                    self._PUBLICKEY_ = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )
            except FileNotFoundError:
                print(
                    f'FileNotFoundError: {key_path}/sinopac_public.pem. Texts will not be encrypted.')
                self._PUBLICKEY_ = None

        if self.DECRYPT:
            try:
                with open(f"{key_path}/sinopac_private.pem", "rb") as f:
                    self._PRIVATEKEY_ = serialization.load_pem_private_key(
                        f.read(), None, backend=default_backend()
                    )
            except FileNotFoundError:
                print(
                    f'FileNotFoundError: {key_path}/sinopac_private.pem. Texts will not be decrypted.')
                self._PRIVATEKEY_ = None

    def encrypt(self, msg: str):
        '''Encrypt text messages.'''
        if self.ENCRYPT and self._PUBLICKEY_:
            return self._PUBLICKEY_.encrypt(
                msg.encode(),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            ).hex()

        print("Can't encrypt without public key")
        return msg

    def decrypt(self, msg: str):
        '''Decrypt text messages.'''
        if self.DECRYPT and self._PRIVATEKEY_:

            if not isinstance(msg, bytes):
                msg = bytes.fromhex(msg)

            return self._PRIVATEKEY_.decrypt(
                msg,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            ).decode()

        print("Can't decrypt without private key")
        return msg


if __name__ == "__main__":

    ct = CipherTool()
    msg = 'Hello World'
    encrypted = ct.encrypt(msg)
    print(encrypted)
