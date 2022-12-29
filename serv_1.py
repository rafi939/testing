# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 12:03:47 2022

@author: user670
"""
'''
import socket
hostname=socket.gethostname()
print(socket.gethostname())
ipadd=socket.gethostbyname(hostname)
print(ipadd)
print(socket.gethostbyaddr(ipadd))
'''
"""
import socket
addr = '127.0.0.25'
try:
    socket.inet_aton(addr)
    print("Valid IP")
except socket.error:
    print("Invalid IP")
    """
'''
import os,ipaddress
os.system('cls') # os.system will clear the console at the start of excution
while(True):
    ip=input("enter ip address: ")
    try:
        print(ipaddress.ip_address(ip))
        print('IP Valid')
    except:
        print('-' *50)
        print('Ip is not valid')
    finally:
        if(ip=='mango'):
            print('Script Finished')
            break
        '''
"""
import os
print(os.system('ipconfig'))
"""
import socket
s=socket.socket()
print("Socket successfully created")
port=40774
s.bind(('',port))
print("socket binded to %s"%(port))
s.listen(5)
print("socket is listening")
while True:
    c,addr=s.accept()
    print('Got connection from',addr)
    c.send('Thank you for connecting'.encode())
    c.close()
    
    
    