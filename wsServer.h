#ifndef WEBSOCKET_SERVER_H
#define WEBSOCKET_SERVER_H

#include <ws.h>

#ifdef __cplusplus
extern "C" {
	#endif

//Function prototypes
void onopen(ws_cli_conn_t client);
void onclose(ws_cli_conn_t client);
void onmessage(ws_cli_conn_t client,const unsigned char *msg,uint64_t size,int type);
void send_joystick_data(const char* message);

#ifdef __cplusplus
}
#endif

#endif //WEBSOCKET_SERVER_H
