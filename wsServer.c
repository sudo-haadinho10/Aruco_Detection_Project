#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ws.h>
#include "wsServer.h"

void onopen(ws_cli_conn_t client)
{
	char *cli,*port;
	cli=ws_getaddress(client);
	port=ws_getport(client);
#ifndef DISABLE_VERBOSE
	printf("Connection opened, addr: %s,port: %s\n",cli,port);
#endif
}

void onclose(ws_cli_conn_t client)
{
	char *cli;
	cli=ws_getaddress(client);
#ifndef DISABLE_VERBOSE
	printf("Connection closed,addr: %s\n",cli);
#endif
}

void onmessage(ws_cli_conn_t client,
	const unsigned char *msg,uint64_t size,int type)
{
	char *cli;
	cli=ws_getaddress(client);
}

void send_joystick_data(const char *message)
{
	ws_sendframe_bcast(9000,message,strlen(message),WS_FR_OP_TXT);
}

