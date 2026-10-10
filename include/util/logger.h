#ifndef LOGGER_H
#define LOGGER_H
#include <stdarg.h>
#include <stdbool.h>
typedef enum {LOG_DEBUG=0,LOG_INFO,LOG_WARNING,LOG_ERROR,LOG_FATAL} LogLevel;
/* Process-global logger, not thread-safe. */
int logger_init(const char *log_folder,const char *log_filename,bool debug);
void logger_cleanup(void);
void logger_set_level(LogLevel level);
void logger_log(LogLevel level,const char *file,int line,const char *fmt,...);
#define LOG_DEBUG(...) logger_log(LOG_DEBUG,__FILE__,__LINE__,__VA_ARGS__)
#define LOG_INFO(...) logger_log(LOG_INFO,__FILE__,__LINE__,__VA_ARGS__)
#define LOG_WARNING(...) logger_log(LOG_WARNING,__FILE__,__LINE__,__VA_ARGS__)
#define LOG_ERROR(...) logger_log(LOG_ERROR,__FILE__,__LINE__,__VA_ARGS__)
#define LOG_FATAL(...) logger_log(LOG_FATAL,__FILE__,__LINE__,__VA_ARGS__)
#endif
