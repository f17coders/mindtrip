package com.a303.notificationms.domain.notification.dto.request;

import lombok.Builder;

@Builder
public record FCMTokenReq(
        String token
) {

}
