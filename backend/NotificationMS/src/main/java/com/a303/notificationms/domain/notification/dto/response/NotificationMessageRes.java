package com.a303.notificationms.domain.notification.dto.response;

import java.time.LocalDateTime;
import lombok.Builder;

@Builder
public record NotificationMessageRes(
	String type,
	String message,
	boolean isWritten,
	LocalDateTime localDateTime
) {

}
