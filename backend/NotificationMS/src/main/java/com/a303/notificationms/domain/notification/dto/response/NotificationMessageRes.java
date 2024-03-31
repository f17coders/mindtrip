package com.a303.notificationms.domain.notification.dto.response;

import lombok.Builder;

@Builder
public record NotificationMessageRes(
	String type,
	String message,
	boolean isWritten
) {

}
