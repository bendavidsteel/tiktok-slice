# 24 hours on TikTok

## Video ID bits

### First 32 bits (0-31)
Unix timestamp in seconds, since unix epoch

### Next 10 bits (32-41)
Milliseconds, 0-999

### Middle 20 bits (42-61)
Vary by content type (video, comment, user etc.)

### Last 2 bits (62-63)
Vary by location of video creation, perhaps associated with a database shard?