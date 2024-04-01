package com.a303.consultms.domain.channel.dto.response;

import com.a303.consultms.domain.member.dto.response.MemberBaseRes;
import com.a303.consultms.domain.message.Message;
import java.util.List;
import lombok.Builder;

@Builder
public record ChannelListRes(
    String channelId,
    int memberId,
    List<Message> messageList
) {

}
