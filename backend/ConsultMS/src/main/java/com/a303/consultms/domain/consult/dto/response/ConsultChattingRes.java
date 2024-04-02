package com.a303.consultms.domain.consult.dto.response;

import lombok.Builder;

@Builder
public record ConsultChattingRes(
    int consultId,
    int memberId,
    String nickname,
    String title,
    String channelId,
    String text,
    boolean shared
) {

}
