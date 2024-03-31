package com.a303.consultms.domain.channel.repository;

import com.a303.consultms.domain.channel.Channel;
import java.util.List;

public interface ChannelCustomRepository {
    List<Channel> findListBySenderOrReceiver(String senderId);
    Channel findBySenderOrReceiver(String senderId, String receiverId);
    List<Channel> findBySender(String senderId);
}
