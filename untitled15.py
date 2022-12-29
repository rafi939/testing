'''from netmiko import ConnectHandler
CSR = {
       "device_type":"cisco_ios",
       #"ip":"sandbox-iosxe-latest-1.cisco.com",
       "ip":"sandbox-iosxe-recomm-1.cisco.com",
       "port":22,
       "username":"developer",
       "password":"C1sco12345"
       }
net_connect = ConnectHandler(**CSR)
#output_runhost = net_connect.send_command('show run | i host')
#output=net_connect.send_command('show ip int brief')
#clock_router=net_connect.send_command('show clock detail')
#route_path=net_connect.send_command('show ip route')
ip_int=net_connect.send_command('sh run')
#print(route_path)
#print(output)
#print(clock_router)
print(ip_int)
net_connect.disconnect()

'''
import socket,sys
try:
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    print("socket successfully created")
except socket.error as err:
    print(err)
port=80
try: 
    host_ip=socket.gethostbyname('www.google.com')
except socket.gaierror: 
    print("there was an error")
    sys.exit()    
s.connect((host_ip,port))
print("socket has successfully connected to google")

"""
from socket import getservbyname, getservbyport
a=getservbyname("ssh") 
b=getservbyport(23)
print(a) 
print(b)
"""